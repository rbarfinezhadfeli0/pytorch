# Documentation: `docs/torch/csrc/jit/tensorexpr/operators/matmul.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/operators/matmul.cpp_docs.md`
- **Size**: 5,108 bytes (4.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/operators/matmul.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/operators/matmul.cpp`
- **Size**: 2,717 bytes (2.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/operators/matmul.h>

namespace torch::jit::tensorexpr {

Tensor computeMatmul(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("matmul", outputShape, dtype);
  const BufHandle a = std::get<BufHandle>(inputs[0]);
  const BufHandle b = std::get<BufHandle>(inputs[1]);

  auto size_a = a.dims();
  auto size_b = b.dims();
  // We currently only support rank 2 matmuls
  TORCH_INTERNAL_ASSERT(size_a.size() == 2 && size_b.size() == 2);
  auto total_size =
      to<LongImm>(IRSimplifier::simplify(
                      cast<int64_t>(size_a[0]) * cast<int64_t>(size_a[1]) *
                      cast<int64_t>(size_b[1]))
                      .node());

  // For small sizes, where N*M*K < 1000, lower matmul to a naive 3-level
  // loopnest. The number is not tuned very carefully, and in future we should
  // fine-tune it as well as we should add more advanced native TE lowerings for
  // matmuls. For bigger sizes we generate a TE ExternalCall, which would call
  // an aten::matmul.
  // Native, even naive, lowering is beneficial when the sizes are small because
  // it allows to eliminate dispatch overhead.
  if (total_size && total_size->value() < 1000) {
    return Reduce(
        "nnc_matmul",
        {size_a[0], size_b[1]},
        Sum(),
        [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
          return Load::make(a, {m, k}) * Load::make(b, {k, n});
        },
        {size_a[1]});
  } else {
    return Tensor(
        ResultBuf.node(),
        ExternalCall::make(ResultBuf, "nnc_aten_matmul", {a, b}, {}));
  }
}

Tensor computeAddMM(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("addmm", outputShape, dtype);
  return Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_addmm",
          {std::get<BufHandle>(inputs[0]),
           std::get<BufHandle>(inputs[1]),
           std::get<BufHandle>(inputs[2])},
          {std::get<int64_t>(inputs[3]),
           std::get<int64_t>(
               inputs[4])})); // TODO: handle other dtypes of alpha and beta
}

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

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

- `torch/csrc/jit/tensorexpr/ir_simplifier.h`
- `torch/csrc/jit/tensorexpr/operators/matmul.h`


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

- **File Documentation**: `matmul.cpp_docs.md`
- **Keyword Index**: `matmul.cpp_kw.md`
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

- **File Documentation**: `matmul.cpp_docs.md_docs.md`
- **Keyword Index**: `matmul.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
